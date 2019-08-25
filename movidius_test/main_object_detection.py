#!/usr/bin/env python

# Modified version of movidius classification-sample.py to test in a snap

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IEPlugin


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", type=str, default='{0}/models/ssd_mobilenet_v2_FP16/ssd_mobilenet_v2.xml')
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files", type=str, nargs="+", default='./images/plate-references/cr-truck-plate1.jpg')
    args.add_argument("-pd", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument("-d", "--device", help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU", default="MYRIAD", type=str)
    args.add_argument("--labels", help="Optional. Path to a labels mapping file", type=str,  default='{0}/models/ssd_mobilenet_v2_FP16/ssd_mobilenet_v2.labels')
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)
    args.add_argument("-ni", "--number_iter", help="Optional. Number of inference iterations", default=1, type=int)
    args.add_argument("-pc", "--perf_counts", help="Optional. Report performance counters", default=False, action="store_true")
    return parser


def main():
    log.basicConfig(format="[%(levelname)s] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    args.model = args.model.format(os.path.dirname(os.path.realpath(__file__)))
    args.labels = args.labels.format(os.path.dirname(os.path.realpath(__file__)))
    print(args.model)
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)

    # Read IR
    log.info("Loading model files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1

    # Read and pre-process input images
    n, c, h, w = net.inputs[input_blob].shape
    image = cv2.imread(args.input)
    original_img = image
    initial_h, initial_w = image.shape[:-1]
    if image.shape[:-1] != (h, w):
        log.warning("Image {} is resized from {} to {}".format(args.input, image.shape[:-1], (h, w)))
        image = cv2.resize(image, (w, h))
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net)

    # Start sync inference
    log.info("Starting inference ({} iterations)".format(args.number_iter))
    infer_time = []
    for i in range(args.number_iter):
        t0 = time()
        res = exec_net.infer(inputs={input_blob: image})
        infer_time.append((time() - t0) * 1000)
    log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))
    if args.perf_counts:
        perf_counts = exec_net.requests[0].get_perf_counts()
        log.info("Performance counters:")
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))
        for layer, stats in perf_counts.items():
            print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'],
                                                              stats['status'], stats['real_time']))

    # Processing output blob
    log.info("Processing output blob")
    res = res[out_blob]
    for obj in res[0][0]:
        # Draw only objects when probability more than specified threshold
        xmin = int(obj[3] * initial_w)
        ymin = int(obj[4] * initial_h)
        xmax = int(obj[5] * initial_w)
        ymax = int(obj[6] * initial_h)
        class_id = int(obj[1])
        if obj[2] > 0.5:
            print("Detected: {0} ({1},{2},{3},{4})".format(class_id, xmin, ymin, xmax, ymax))
            color = (0, 255, 0)
            try:
                cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), color, 2)
            except:
                pass
        # else:
        #     try:
        #         color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
        #         cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), color, 2)
        #     except:
        #         pass

        cv2.imwrite('./output.jpg', original_img)


if __name__ == '__main__':
    sys.exit(main() or 0)
