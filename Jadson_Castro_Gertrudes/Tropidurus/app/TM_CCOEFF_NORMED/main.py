import cv2 as cv
import os
import tempfile
import shutil


def save_first_frame(frames_dir_path):
    videos_path = [f"../videos/{filename}" for filename in os.listdir("../videos")]
    for path in videos_path:
        video = cv.VideoCapture(path)
        ok, frame = video.read()
        cv.imwrite(f"{frames_dir_path}/{path.split('/')[2].split('.')[0]}.jpg", frame)


def convert_coordinates(size, box):
    x_min = abs(int(size[0] * (box[0] - box[2])))
    y_min = abs(int(size[1] * (box[1] - box[3])))
    x_max = abs(int(size[0] * (box[0] + box[2])))
    y_max = abs(int(size[1] * (box[1] + box[3])))
    return x_min, y_min, x_max, y_max


def get_templates(darknet_labels, filename, frames_dir_path):
    templates = []
    for line in darknet_labels:
        img = cv.imread(f"{frames_dir_path}/{filename.split('.')[0]}.jpg")
        height, width, _ = img.shape
        x = line.split(" ")[1]
        y = line.split(" ")[2]
        w = line.split(" ")[3]
        h = line.split(" ")[4]
        box = (float(x), float(y), float(w), float(h))
        size = (width, height)
        x_min, y_min, x_max, y_max = convert_coordinates(size, box)
        template = img[y_min:y_max, x_min:x_max]
        templates.append(template)
    return templates


def template_matching(frames_dir_path, results_dir_path):
    for filename in os.listdir("labels"):
        with open(f"labels/{filename}") as label_file:
            templates = get_templates(label_file.readlines(), filename, frames_dir_path)
            video_path = f"../videos/{filename.split('.')[0]}.mp4"
            video = cv.VideoCapture(video_path)
            width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
            video_output_path = f"{results_dir_path}/{filename.split('.')[0]}_results.mp4"
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


os.chdir("TM_CCOEFF_NORMED/")
frames_dir_path = tempfile.mkdtemp()
results_dir_path = "results"
if not os.path.exists(results_dir_path):
    os.mkdir(results_dir_path)

save_first_frame(frames_dir_path)
template_matching(frames_dir_path, results_dir_path)

shutil.rmtree(frames_dir_path)
