import tensorflow as tf


class Viz(object):
    def __init__(self, global_config):
        self.BATCH_SIZE = global_config.trainer.BATCH_SIZE
        self.ZENITH_LEVEL = global_config.dataset.ZENITH_LEVEL
        self.AZIMUTH_LEVEL = global_config.dataset.AZIMUTH_LEVEL
        
    def _register_graph(self):
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


def _add_summary_ops(self):
    """Add extra summary operations."""
    mc = self.mc

    iou_summary_placeholders = []
    iou_summary_ops = []

    for cls in mc.CLASSES:
        ph = tf.placeholder(tf.float32, name=cls+'_iou')
        iou_summary_placeholders.append(ph)
        iou_summary_ops.append(
            tf.summary.scalar('Eval/'+cls+'_iou', ph,
                              collections='eval_summary')
        )

    self.iou_summary_placeholders = iou_summary_placeholders
    self.iou_summary_ops = iou_summary_ops
