import os
import editdistance
import json
from PIL import Image
import glob


class AnswerFinder(object):
    def __init__(self, space_tolerance=True, period_comma_tolerance=True, global_distance=1,
                 minimal_length_for_fuzzy_match=5):
        self.space_tolerance = space_tolerance
        self.period_comma_tolerance = period_comma_tolerance
        self.global_distance = global_distance
        self.minimal_length_for_fuzzy_match = minimal_length_for_fuzzy_match

    def find(self, answer, words, from_beginning):
        if answer == "":
            return None, None, None
        answer_tokens = answer.split()
        end_index = None
        start_index = None
        # reverse the search order because the total is typically in the end
        if answer_tokens[0] in words:
            if from_beginning:
                start_index = min(loc for loc, val in enumerate(words) if val == answer_tokens[0])
            else:
                start_index = max(loc for loc, val in enumerate(words) if val == answer_tokens[0])
        if answer_tokens[-1] in words:
            if from_beginning:
                end_index = min(loc for loc, val in enumerate(words) if val == answer_tokens[-1])
            else:
                end_index = max(loc for loc, val in enumerate(words) if val == answer_tokens[-1])
        if start_index is not None and end_index is not None:
            if start_index > end_index:
                if answer_tokens[-1] in words[start_index:]:
                    end_index = words[start_index:].index(answer_tokens[-1])
                    end_index += start_index
                else:
                    # Last try
                    start_index, end_index, extracted_answer = self.harsh_find(answer_tokens, words)
                    return start_index, end_index, extracted_answer

            assert start_index <= end_index
            extracted_answer = ' '.join(words[start_index:end_index + 1])
            if not self._two_answers_are_equal(answer, extracted_answer):
                start_index, end_index, extracted_answer = self.harsh_find(answer_tokens, words)
                return start_index, end_index, extracted_answer
            else:
                return start_index, end_index, extracted_answer

        else:
            start_index, end_index, extracted_answer = self.harsh_find(answer_tokens, words)
            return start_index, end_index, extracted_answer

    def _two_answers_are_equal(self, answer1, answer2):
        if self.space_tolerance:
            answer1 = answer1.replace(' ', '')
            answer2 = answer2.replace(' ', '')
        if self.period_comma_tolerance:
            answer1 = answer1.replace('.', ',')
            answer2 = answer2.replace('.', ',')
        if len(answer1) >= self.minimal_length_for_fuzzy_match:
            return editdistance.eval(answer1, answer2) <= self.global_distance
        return answer1 == answer2

    def reverse_harsh_find(self, answer_tokens, words):
        answer_raw = ''.join(answer_tokens)
        answer = ' '.join(answer_tokens)
        for (ind, w) in enumerate(words):
            if answer_raw.endswith(w):  # Looks like words are split
                for inc in range(1, len(answer_tokens) + 5):
                    if ind - inc < 0:
                        break
                    w = words[ind - inc] + w
                    start_index = ind - inc
                    end_index = ind
                    ext_list = words[start_index:end_index + 1]
                    extracted_answer = ' '.join(ext_list)

                    if self._two_answers_are_equal(answer, extracted_answer):
                        return start_index, end_index, extracted_answer
        return None, None, None

    def harsh_find(self, answer_tokens, words):
        answer_raw = ''.join(answer_tokens)
        answer = ' '.join(answer_tokens)
        if len(answer_tokens) == 1:
            for (ind, w) in enumerate(words):
                if self._two_answers_are_equal(answer, w):
                    start_index = end_index = ind
                    return start_index, end_index, w
        for (ind, w) in enumerate(words):
            if answer_raw.startswith(w):  # Looks like words are split
                for inc in range(1, len(answer_tokens) + 5):
                    if ind + inc >= len(words):
                        break
                    w = w + words[ind + inc]
                    start_index = ind
                    end_index = ind + inc
                    ext_list = words[start_index:end_index + 1]
                    extracted_answer = ' '.join(ext_list)
                    if self._two_answers_are_equal(answer, extracted_answer):
                        return start_index, end_index, extracted_answer
        return self.reverse_harsh_find(answer_tokens, words)


def bbox_normalized(box, width, height):
    return [int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height))]


def bbox_denormalized(box, width, height):
    return [int(width * (box[0] / 1000)),
            int(height * (box[1] / 1000)),
            int(width * (box[2] / 1000)),
            int(height * (box[3] / 1000))]


def convert_one_file(file_name, key, ocr_dir, data_dir, image_dir, answer_finder=AnswerFinder()):
    all_words = []
    ocr_path = os.path.join(ocr_dir, file_name + ".txt")
    with open(ocr_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if not line == "\n":
            splits = line.split(",")
            assert len(splits) >= 9
            text = ",".join(splits[8:])
            x1, y1, x2, y2, x3, y3, x4, y4 = splits[:8]
            all_words.append({
                "text": text,
                "boundingBox": [x1, y1, x2, y2, x3, y3, x4, y4]
            })

    image_path = os.path.join(image_dir, file_name + ".jpg")
    image = Image.open(image_path)
    width, height = image.size

    all_tokens = []
    for word in all_words:
        tokens = word['text'].split()
        x1, y1, x2, y2, x3, y3, x4, y4 = word['boundingBox']

        new_x1 = min([int(x1), int(x2), int(x3), int(x4)])
        new_x2 = max([int(x1), int(x2), int(x3), int(x4)])
        new_y1 = min([int(y1), int(y2), int(y3), int(y4)])
        new_y2 = max([int(y1), int(y2), int(y3), int(y4)])
        x_length_per_word = (new_x2 - new_x1) / len(tokens)

        for idx, token in enumerate(tokens):
            min_x = int(new_x1 + idx * x_length_per_word)
            max_x = int(new_x1 + (idx + 1) * x_length_per_word)
            all_tokens.append({
                "text": token,
                "position": bbox_normalized([min_x, new_y1, max_x, new_y2], width, height)
            })

    label_path = os.path.join(data_dir, file_name + ".txt")
    with open(label_path, "r", encoding="utf8") as f:
        data = json.load(f)

    labels = []
    ground_truth = [0, 0, 0, 0]
    text = ""
    if key in data.keys():
        label = data[key]
        text = label
        start, end, _ = answer_finder.find(label, [word['text'] for _, word in enumerate(all_tokens)], key != "total")
        if start is not None and end is not None:
            labels = list(range(start + 1, end + 2))
            min_x = min((all_tokens[i-1]["position"][0] for i in labels))
            min_y = min((all_tokens[i-1]["position"][1] for i in labels))
            max_x = max((all_tokens[i-1]["position"][2] for i in labels))
            max_y = max((all_tokens[i-1]["position"][3] for i in labels))
            ground_truth = [min_x, min_y, max_x, max_y]

    return all_tokens, labels, ground_truth, text


def generate_documents(key, ocr_dir, data_dir, image_dir):
    documents = {}
    label_files = glob.glob(os.path.join(data_dir, "*.txt"))
    for i_file, label_file in enumerate(label_files):
        file_name = os.path.splitext(os.path.basename(label_file))[0]
        documents[i_file] = {}
        documents[i_file]["image_path"] = os.path.join(image_dir, file_name + ".jpg")
        ocr, labels, gt, text = convert_one_file(file_name, key, ocr_dir, data_dir, image_dir)
        documents[i_file]["OCR"] = ocr
        documents[i_file]["labels"] = labels
        documents[i_file]["ground_truth"] = gt
        documents[i_file][key] = text

    return documents


