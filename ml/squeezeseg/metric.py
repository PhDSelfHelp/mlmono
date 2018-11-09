import tensorflow as tf

from ml.base import StepMetric

class Viz(StepMetric):
    @classmethod
    def from_config(cls, global_config, metric_config):
        return Viz(global_config)

    def __init__(self, global_config):
        self.NUM_CLASS = global_config.io.num_class
        self.BATCH_SIZE = global_config.trainer.batch_size
        self.ZENITH_LEVEL = global_config.io.zenith_level
        self.AZIMUTH_LEVEL = global_config.io.azimuth_level
        
    def register_to_graph(self, graph):
        """Define the visualization operation."""
        BATCH_SIZE = self.BATCH_SIZE
        ZENITH_LEVEL = self.ZENITH_LEVEL
        AZIMUTH_LEVEL = self.AZIMUTH_LEVEL

        self.label_to_show = tf.placeholder(
            tf.float32, [None, ZENITH_LEVEL, AZIMUTH_LEVEL, 3],
            name='label_to_show'
        )
        self.depth_image_to_show = tf.placeholder(
            tf.float32, [None, ZENITH_LEVEL, AZIMUTH_LEVEL, 1],
            name='depth_image_to_show'
        )
        self.pred_image_to_show = tf.placeholder(
            tf.float32, [None, ZENITH_LEVEL, AZIMUTH_LEVEL, 3],
            name='pred_image_to_show'
        )
        self.show_label = tf.summary.image('label_to_show',
                                           self.label_to_show, collections='image_summary',
                                           max_outputs=BATCH_SIZE)
        self.show_depth_img = tf.summary.image('depth_image_to_show',
                                               self.depth_image_to_show, collections='image_summary',
                                               max_outputs=BATCH_SIZE)
        self.show_pred = tf.summary.image('pred_image_to_show',
                                          self.pred_image_to_show, collections='image_summary',
                                          max_outputs=BATCH_SIZE)


class IOUSummary(StepMetric):
    @classmethod
    def from_config(cls, global_config, metric_config):
        return IOUSummary(global_config)

    def __init__(self, global_config):
        self.global_config = global_config
        self.iou_summary_ops = None

    def register_to_graph(self, graph):
        iou_summary_placeholders = []
        iou_summary_ops = []
        for obj_cls in self.global_config.io.classes:
            ph = tf.placeholder(tf.float32, name=obj_cls+'_iou')
            iou_summary_placeholders.append(ph)
            iou_summary_ops.append(
                tf.summary.scalar('Eval/{}_iou'.format(obj_cls),
                                ph,
                                collections='eval_summary')
            )
        self.iou_summary_placeholders = iou_summary_placeholders
        self.iou_summary_ops = iou_summary_ops

        # Run evaluation on the batch
        ious, _, _, _ = evaluate_iou(label_per_batch,
                                     pred_cls * np.squeeze(lidar_mask_per_batch),
                                     self.NUM_CLASS)

    @staticmethod
    def _evaluate_iou(label, pred, n_class, epsilon=1e-12):
        """Evaluation script to compute pixel level IoU.

        Args:
        label: N-d array of shape [batch, W, H], where each element is a class
            index.
        pred: N-d array of shape [batch, W, H], the each element is the predicted
            class index.
        n_class: number of classes
        epsilon: a small value to prevent division by 0

        Returns:
        IoU: array of lengh n_class, where each element is the average IoU for this
            class.
        tps: same shape as IoU, where each element is the number of TP for each
            class.
        fps: same shape as IoU, where each element is the number of FP for each
            class.
        fns: same shape as IoU, where each element is the number of FN for each
            class.
        """

        assert label.shape == pred.shape, \
            'label and pred shape mismatch: {} vs {}'.format(
                label.shape, pred.shape)

        ious = np.zeros(n_class)
        tps = np.zeros(n_class)
        fns = np.zeros(n_class)
        fps = np.zeros(n_class)

        for cls_id in range(n_class):
            tp = np.sum(pred[label == cls_id] == cls_id)
            fp = np.sum(label[pred == cls_id] != cls_id)
            fn = np.sum(pred[label == cls_id] != cls_id)

            ious[cls_id] = tp / (tp + fn + fp + epsilon)
            tps[cls_id] = tp
            fps[cls_id] = fp
            fns[cls_id] = fn

        return ious, tps, fps, fns