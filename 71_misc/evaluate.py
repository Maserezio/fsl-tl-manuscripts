import argparse
import os
from data.dataset import PATH_DICT
from evaluate_util import udiads_textline_evaluate



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default='out')
    parser.add_argument("--gt_dir", type=str, default='/data/databases/maps/text_icdar')
    args = parser.parse_args()

    for dataset in ['latin1', 'latin2',  'syr']:
        print('Evaluating', dataset)
        result_path = os.path.join(args.result_dir, PATH_DICT[dataset])
        results = {}
        base_path = os.path.join(args.gt_dir, PATH_DICT[dataset])
        for split in ['validation']:
                gt_path = os.path.join(base_path, f'text-line-gt-{PATH_DICT[dataset]}', split)

                Pixel_IU, Line_IU, DR, RA, FM = udiads_textline_evaluate(result_directory=result_path, gt_directory=gt_path)

                print("FINAL RESULTS")
                print("Pixel IU: ", Pixel_IU)
                print("Line IU: ", Line_IU)
                print("Detection Rate: ", DR)
                print("Recognition Accuracy: ", RA)
                print("F-measure: ", FM)
        print()
    print("Done")
