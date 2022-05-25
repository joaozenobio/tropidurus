import cv2 as cv
import os
import json
import tempfile
import shutil


def save_first_frame(frames_dir_path):
    videos_path = [f"../videos/{filename}" for filename in os.listdir("../videos")]
    for path in videos_path:
        video = cv.VideoCapture(path)
        ok, frame = video.read()
        cv.imwrite(f"{frames_dir_path}/{path.split('/')[2].split('.')[0]}.jpg", frame)


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


def get_templates_from_frame_json(frame):
    templates = []
    for i in range(len(frame['objects'])):
        img = cv.imread(frame['filename'])
        height, width, _ = img.shape
        x = frame['objects'][i]['relative_coordinates']['center_x']
        y = frame['objects'][i]['relative_coordinates']['center_y']
        w = frame['objects'][i]['relative_coordinates']['width']
        h = frame['objects'][i]['relative_coordinates']['height']
        box = (float(x), float(y), float(w), float(h))
        size = (width, height)
        x_min, y_min, x_max, y_max = convert_coordinates(size, box)
        template = img[y_min:y_max, x_min:x_max]
        templates.append(template)
    return templates


def template_matching_output_json(frames_dir_path, resultados_dir_path):
    with open('output.json') as output_file:
        data = json.load(output_file)
        for frame in data:
            templates = get_templates_from_frame_json(frame)
            filename = "../videos/" + frame['filename'].split('/')[-1].split('.')[0] + ".mp4"
            video = cv.VideoCapture(filename)
            width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
            video_output_path = f"{resultados_dir_path}/{filename.split('/')[2].split('.')[0]}_result.mp4"
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            video_writer = cv.VideoWriter(video_output_path, fourcc, 30.0, (width, height))
            # Jump first frame
            ok, frame = video.read()
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                for template in templates:
                    result = cv.matchTemplate(frame, template, cv.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
                    _, template_width, template_height = template.shape[::-1]
                    bottom_right = (max_loc[0] + template_width, max_loc[1] + template_height)
                    cv.rectangle(frame, max_loc, bottom_right, 255, 2)
                video_writer.write(frame)
            video.release()
            video_writer.release()

os.chdir(os.getcwd() + "/YOLOv4-TM_CCOEFF_NORMED/")
frames_dir_path = tempfile.mkdtemp()
results_dir_path = "results"
if not os.path.exists(results_dir_path):
    os.mkdir(results_dir_path)

save_first_frame(frames_dir_path)
darknet(frames_dir_path)
template_matching_output_json(frames_dir_path, results_dir_path)

shutil.rmtree(frames_dir_path)
